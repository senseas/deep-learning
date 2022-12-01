package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ObjectCreationExpression;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.expression.TypeParametersExpression;
import com.deep.framework.ast.lexer.Token;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;
import java.util.stream.Collectors;

@Data
public class MethodDeclaration extends Declaration {
    private List<TokenType> modifiers;
    private Type returnValue;
    private Name name;
    private ParametersExpression parameters;
    private BlockStatement body;

    public MethodDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (!(node.getPrarent() instanceof ClassOrInterfaceDeclaration || node.getPrarent() instanceof EnumDeclaration || node.getPrarent() instanceof ObjectCreationExpression)) return;
        if (node instanceof MethodDeclaration) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (b instanceof BlockStatement) {
                Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                    if (m instanceof Name && n instanceof ParametersExpression) {
                        MethodDeclaration declare = new MethodDeclaration(node.getPrarent());
                        List<TokenType> modifiers = a.getMethodModifiers();
                        declare.setModifiers(modifiers);
                        declare.setName((Name) m);
                        declare.setParameters((ParametersExpression) n);
                        declare.setBody((BlockStatement) b);
                        declare.setChildrens(a.getChildrens());
                        declare.getChildrens().add(b);
                        TypeParametersExpression.parser(declare);
                        node.replaceAndRemove(a, declare, b);
                    }
                });
            }
        });
    }
}