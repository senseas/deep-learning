package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.ENUM;
import static com.deep.framework.ast.lexer.TokenType.IMPLEMENTS;

@Data
public class EnumDeclaration extends Declaration {
    private List<Object> implementedTypes;
    private List<TokenType> modifiers;
    private Name name;
    private ParametersExpression parameters;
    private BlockStatement body;

    public EnumDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        PackageDeclaration.parser(node);
        ImportDeclaration.parser(node);

        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (b instanceof BlockStatement) {
                Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                    if (m.equals(ENUM)) {
                        EnumDeclaration declare = new EnumDeclaration(node.getPrarent());
                        List<TokenType> modifiers = a.getFieldModifiers();
                        b.setPrarent(declare);
                        declare.setModifiers(modifiers);
                        declare.setName((Name) n);
                        declare.setBody((BlockStatement) b);
                        declare.setChildrens(a.getChildrens());
                        declare.getChildrens().add(b);
                        a.getChildrens().remove(m);

                        node.replaceAndRemove(a, declare, b);

                        parserImplements(declare);

                        EnumConstantDeclaration.parser(b);
                        MethodDeclaration.parser(b);
                        FieldDeclaration.parser(b);
                    }
                });
            }
        });
    }

    public static void parserImplements(EnumDeclaration classDeclare) {
        if (classDeclare.getChildrens().contains(IMPLEMENTS)) {
            List<Object> list = null;
            for (Object a : List.copyOf(classDeclare.getChildrens())) {
                if (a instanceof BlockStatement) {
                    classDeclare.setImplementedTypes(list);
                    classDeclare.getChildrens().removeAll(list);
                    return;
                } else if (a.equals(IMPLEMENTS)) {
                    list = new ArrayList<>();
                    classDeclare.getChildrens().remove(a);
                } else if (Objects.nonNull(list)) {
                    list.add(a);
                }
            }
        }
    }

}