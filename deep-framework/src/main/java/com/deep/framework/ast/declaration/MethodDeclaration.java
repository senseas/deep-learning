package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
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
    private ParametersExpression parameters;
    private Name name;
    private BlockStatement body;

    public MethodDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof MethodDeclaration) return;
        List.copyOf(node.getChildrens()).stream().map(a -> (Node) a).reduce((a, b) -> {
            if (b instanceof BlockStatement) {
                Object m = a.getChildrens().get(a.getChildrens().size() - 2);
                Object n = a.getChildrens().get(a.getChildrens().size() - 1);
                if (m instanceof Name && n instanceof ParametersExpression && b instanceof BlockStatement) {
                    MethodDeclaration methodDeclare = new MethodDeclaration(node.getPrarent());
                    List<TokenType> modifiers = a.getChildrens().stream().filter(e -> Method_Modifiers.contains(e)).map(o -> ((Token) o).getTokenType()).collect(Collectors.toList());
                    a.getChildrens().removeAll(modifiers);

                    methodDeclare.setModifiers(modifiers);
                    methodDeclare.setName((Name) m);
                    methodDeclare.setParameters((ParametersExpression) n);
                    methodDeclare.setBody((BlockStatement) b);
                    methodDeclare.setChildrens(a.getChildrens());
                    methodDeclare.getChildrens().add(b);
                    TypeParametersExpression.parser(methodDeclare);
                    node.replaceAndRemove(a, methodDeclare, b);
                }
            }
            return b;
        });
    }
}
