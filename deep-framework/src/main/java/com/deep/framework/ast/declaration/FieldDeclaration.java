package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.lexer.Token;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.statement.Statement;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;
import java.util.stream.Collectors;

@Data
public class FieldDeclaration extends Declaration {
    private List<TokenType> modifiers;
    private Type type;
    private Name name;

    public FieldDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (!(node.getPrarent() instanceof ClassOrInterfaceDeclaration)) return;
        if (node instanceof FieldDeclaration) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (!(b instanceof BlockStatement)) {
                if (a instanceof Statement) {
                    FieldDeclaration declare = new FieldDeclaration(node.getPrarent());
                    List<Node> modifiers = a.getChildrens().stream().filter(e -> Field_Modifiers.contains(e.getTokenType())).toList();
                    a.getChildrens().removeAll(modifiers);

                    declare.setModifiers(modifiers.stream().map(Token::getTokenType).collect(Collectors.toList()));
                    declare.setType(Type.getType(a.getChildrens().first()));
                    declare.setChildrens(a.getChildrens());
                    node.replace(a, declare);
                }
            }
        });
    }
}
